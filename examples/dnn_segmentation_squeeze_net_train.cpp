// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This program was used to train the resnet34_1000_imagenet_classifier.dnn
    network used by the dnn_imagenet_ex.cpp example program.

    You should be familiar with dlib's DNN module before reading this example
    program.  So read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp first.
*/



#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>
#include <iterator>
#include <thread>
#include <dlib/xml_parser.h>
#include <fstream>

using namespace std;
using namespace dlib;

struct image_info
{
    string image_filename;
    string label_filename;
};

std::vector<image_info> get_pascal_voc2012_listing(
    const std::string& voc2012_folder,
    const std::string& file = "train" // "train", "trainval", or "val"
)
{
    std::ifstream in(voc2012_folder + "/ImageSets/Segmentation/" + file + ".txt");

    std::vector<image_info> results;

    while (in) {
        std::string basename;
        in >> basename;

        if (!basename.empty()) {
            image_info image_info;
            image_info.image_filename = voc2012_folder + "/JPEGImages/" + basename + ".jpg";
            image_info.label_filename = voc2012_folder + "/SegmentationClass/" + basename + ".png";
            results.push_back(image_info);
        }
    }

    return results;
}

std::vector<image_info> get_pascal_voc2012_train_listing(
    const std::string& voc2012_folder
)
{
    return get_pascal_voc2012_listing(voc2012_folder, "trainval");
}

typedef std::pair<matrix<rgb_pixel>, matrix<uint16_t>> training_sample;

inline bool operator == (const dlib::rgb_pixel& a, const dlib::rgb_pixel& b)
{
    return a.red == b.red && a.green == b.green && a.blue == b.blue;
}

// ----------------------------------------------------------------------------------------

struct Voc2012class {
    Voc2012class(uint16_t index, const dlib::rgb_pixel& rgb_label, const char* classlabel)
        : index(index), rgb_label(rgb_label), classlabel(classlabel)
    {}

    const uint16_t index = 0;
    const dlib::rgb_pixel rgb_label;
    const char* classlabel = nullptr;
};

namespace {
    constexpr int class_count = 21; // background + 20 classes

    const std::vector<Voc2012class> classes = {
        Voc2012class(0, dlib::rgb_pixel(0, 0, 0), ""), // background

        // The cream-colored `void' label is used in border regions and to mask difficult objects
        // (see http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html)
        Voc2012class(dlib::loss_multiclass_log_per_pixel_::label_to_ignore,
            dlib::rgb_pixel(224, 224, 192), "border"),

        Voc2012class(1,  dlib::rgb_pixel(128,   0,   0), "aeroplane"),
        Voc2012class(2,  dlib::rgb_pixel(  0, 128,   0), "bicycle"),
        Voc2012class(3,  dlib::rgb_pixel(128, 128,   0), "bird"),
        Voc2012class(4,  dlib::rgb_pixel(  0,   0, 128), "boat"),
        Voc2012class(5,  dlib::rgb_pixel(128,   0, 128), "bottle"),
        Voc2012class(6,  dlib::rgb_pixel(  0, 128, 128), "bus"),
        Voc2012class(7,  dlib::rgb_pixel(128, 128, 128), "car"),
        Voc2012class(8,  dlib::rgb_pixel( 64,   0,   0), "cat"),
        Voc2012class(9,  dlib::rgb_pixel(192,   0,   0), "chair"),
        Voc2012class(10, dlib::rgb_pixel( 64, 128,   0), "cow"),
        Voc2012class(11, dlib::rgb_pixel(192, 128,   0), "diningtable"),
        Voc2012class(12, dlib::rgb_pixel( 64,   0, 128), "dog"),
        Voc2012class(13, dlib::rgb_pixel(192,   0, 128), "horse"),
        Voc2012class(14, dlib::rgb_pixel( 64, 128, 128), "motorbike"),
        Voc2012class(15, dlib::rgb_pixel(192, 128, 128), "person"),
        Voc2012class(16, dlib::rgb_pixel(  0,  64,   0), "pottedplant"),
        Voc2012class(17, dlib::rgb_pixel(128,  64,   0), "sheep"),
        Voc2012class(18, dlib::rgb_pixel(  0, 192,   0), "sofa"),
        Voc2012class(19, dlib::rgb_pixel(128, 192,   0), "train"),
        Voc2012class(20, dlib::rgb_pixel(  0,  64, 128), "tvmonitor"),
    };
}

template <typename Predicate>
const Voc2012class& find_voc2012_class(Predicate predicate)
{
    const auto i = std::find_if(classes.begin(), classes.end(), predicate);

    if (i != classes.end()) {
        return *i;
    }
    else {
        throw std::runtime_error("Unable to find a matching VOC2012 class");
    }
}
// ----------------------------------------------------------------------------------------

template <typename SUBNET> using fire_expand_a1 = relu<con<64, 1, 1, 1, 1, 1, 1, SUBNET>>;
template <typename SUBNET> using fire_expand_a2 = relu<con<64, 3, 3, 1, 1, 1, 1, SUBNET>>;
template <typename SUBNET> using fire_squeeze_a = inception2<fire_expand_a1, fire_expand_a2, SUBNET>;

template <typename SUBNET> using fire_expand_b1 = relu<con<128, 1, 1, 1, 1, 1, 1, SUBNET>>;
template <typename SUBNET> using fire_expand_b2 = relu<con<128, 3, 3, 1, 1, 1, 1, SUBNET>>;
template <typename SUBNET> using fire_squeeze_b = inception2<fire_expand_b1, fire_expand_b2, SUBNET>;

template <typename SUBNET> using fire_expand_c1 = relu<con<192, 1, 1, 1, 1, 1, 1, SUBNET>>;
template <typename SUBNET> using fire_expand_c2 = relu<con<192, 3, 3, 1, 1, 1, 1, SUBNET>>;
template <typename SUBNET> using fire_squeeze_c = inception2<fire_expand_c1, fire_expand_c2, SUBNET>;

template <typename SUBNET> using fire_expand_d1 = relu<con<256, 1, 1, 1, 1, 1, 1, SUBNET>>;
template <typename SUBNET> using fire_expand_d2 = relu<con<256, 3, 3, 1, 1, 1, 1, SUBNET>>;
template <typename SUBNET> using fire_squeeze_d = inception2<fire_expand_d1, fire_expand_d2, SUBNET>;
template <int N, typename SUBNET> using dilate = con < 256, 3, 3, 1, 1, N, N, skip4<SUBNET>>;

template <typename SUBNET> using refine_expand_a1 = relu <con<128, 3, 3, 1, 1, 1, 1, skip3<SUBNET>>>;
template <typename SUBNET> using refine_expand_a2 = cont<128,3,3,2,2, SUBNET>;
template <typename SUBNET> using refine_a = inception2<refine_expand_a1, refine_expand_a2, SUBNET>;

template <typename SUBNET> using refine_expand_b1 = relu <con<64, 3, 3, 1, 1, 1, 1, skip2<SUBNET>>>;
template <typename SUBNET> using refine_expand_b2 = cont<64, 4, 4, 2, 2, SUBNET>;
template <typename SUBNET> using refine_b = inception2<refine_expand_b1, refine_expand_b2, SUBNET>;

template <typename SUBNET> using refine_expand_c1 = relu <con<64, 3, 3, 1, 1, 1, 1, skip1<SUBNET>>>;
template <typename SUBNET> using refine_expand_c2 = cont<64, 3, 3, 2, 2, SUBNET>;
template <typename SUBNET> using refine_c = inception2<refine_expand_c1, refine_expand_c2, SUBNET>;



// training network type


using net_type = dlib::loss_multiclass_log_per_pixel<
                                                    bn_con<cont<21,3,3,2,2,
                                                    relu<bn_con<con<64,3,3,1,1,1,1,refine_c<
                                                    relu<bn_con<con<64,3,3,1,1,1,1,refine_b<
                                                    relu<bn_con<con<128,3,3,1,1,1,1,refine_a<
                                                    add_prev7<bn_con<con<256,3,3,1,1,4,4,skip4<
                                                    tag7<add_prev6<bn_con<con<256,3,3,1,1,3,3,skip4<
                                                    tag6<add_prev5<bn_con<con<256,3,3,1,1,2,2,skip4< 
													tag5<bn_con<con<256,3,3,1,1,1,1, tag4<
													fire_squeeze_d<relu<bn_con<con<64, 1, 1, 1, 1, 1, 1,
													fire_squeeze_d<relu<bn_con<con<64, 1, 1, 1, 1, 1, 1,
													fire_squeeze_c<relu<bn_con<con<48, 1, 1, 1, 1, 1, 1,
													fire_squeeze_c<relu<bn_con<con<48, 1, 1, 1, 1, 1, 1,
													max_pool<3, 3, 2, 2, tag3<
													fire_squeeze_b<relu<bn_con<con<32,1,1,1,1,1,1,
													fire_squeeze_b<relu<bn_con<con<32,1,1,1,1,1,1,
													max_pool<3, 3, 2, 2, tag2<
													fire_squeeze_a<relu<bn_con<con<16,1,1,1,1,1,1,
													fire_squeeze_a<relu<bn_con<con<16,1,1,1,1,1,1,
                                                    max_pool<3, 3, 2, 2, tag1<relu<bn_con<con<64,3,3,2,2,1,1,
                                                    dlib::input<dlib::matrix<dlib::rgb_pixel>>>>>>>>
                                                    >>>>
                                                    >>>>
                                                    >>>>
                                                    >>>
                                                    >>>
                                                    >>>
                                                    >>>
                                                    >>>
                                                    >>                                                    
                                                    >>>
                                                    >>>
                                                    >>
                                                    >>>
                                                    >>>
                                                    >>>
                                                    >>>
                                                    >>>
                                                    >>>>>>>>>>>>>>>>;

// ---------------------------------------------------------------------------------------

rectangle make_random_cropping_rect_resnet(
    const matrix<rgb_pixel>& img,
    dlib::rand& rnd
)
{
    // figure out what rectangle we want to crop from the image
    auto size = 227;
    rectangle rect(size, size);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
                 rnd.get_random_32bit_number()%(img.nr()-rect.height()));

    return move_rect(rect, offset);
}

// ----------------------------------------------------------------------------------------

void randomly_crop_image (
    const matrix<rgb_pixel>& input_image,
    const matrix<uint16_t>& label_image,
    training_sample& crop,
    dlib::rand& rnd
)
{
    const auto rect = make_random_cropping_rect_resnet(input_image, rnd);

    const chip_details chip_details(rect, chip_dims(227, 227));

    // Crop the input image.
    extract_image_chip(input_image, chip_details, crop.first);

    // Crop the labels correspondingly. However, note that here bilinear
    // interpolation would make absolutely no sense.
    extract_image_chip(label_image, chip_details, crop.second);

    // Also randomly flip the input image and the labels.
    if (rnd.get_random_double() > 0.5) {
        crop.first = fliplr(crop.first);
        crop.second = fliplr(crop.second);
    }

    // And then randomly adjust the colors.
//    apply_random_color_offset(crop.first, rnd);
    
}

const Voc2012class& find_voc2012_class(const dlib::rgb_pixel& rgb_label)
{
    return find_voc2012_class(
        [&rgb_label](const Voc2012class& voc2012class) {
            return rgb_label == voc2012class.rgb_label;
        }
    );
}

inline uint16_t rgb_label_to_index_label(const dlib::rgb_pixel& rgb_label)
{
    return find_voc2012_class(rgb_label).index;
}

void rgb_label_image_to_index_label_image(const dlib::matrix<dlib::rgb_pixel>& rgb_label_image, 
                                          dlib::matrix<uint16_t>& index_label_image)
{
    const long nr = rgb_label_image.nr();
    const long nc = rgb_label_image.nc();

    index_label_image.set_size(nr, nc);

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            index_label_image(r, c) = rgb_label_to_index_label(rgb_label_image(r,c));
        }
    }
}

// --------------------------------------------------------------------------------------


int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        return 1;
    }

    cout << "\nSCANNING PASCAL VOC2012 DATASET\n" << endl;

    const auto listing = get_pascal_voc2012_train_listing(argv[1]);
    cout << "images in dataset: " << listing.size() << endl;
    if (listing.size() == 0)
    {
        cout << "Didn't find the VOC2012 dataset. " << endl;
        return 1;
    }


    set_dnn_prefer_smallest_algorithms();


    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    net_type net;

 //   cout << net << "\n";
    dnn_trainer<net_type> trainer(net,sgd(weight_decay, momentum));
    trainer.be_verbose();
    trainer.set_learning_rate(initial_learning_rate);
    trainer.set_synchronization_file("squeeesegnet_voc2012_trainer_state_file.dat", std::chrono::minutes(10));
    // This threshold is probably excessively large.  You could likely get good results
    // with a smaller value but if you aren't in a hurry this value will surely work well.
    trainer.set_iterations_without_progress_threshold(20000);
    // Since the progress threshold is so large might as well set the batch normalization
    // stats window to something big too.
    set_all_bn_running_stats_window_sizes(net, 1000);

    std::vector<matrix<rgb_pixel>> samples;
    std::vector<matrix<uint16_t>> labels;

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<training_sample> data(120);
    auto f = [&data, &listing](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        matrix<rgb_pixel> input_image;
        matrix<rgb_pixel> rgb_label_image;
        matrix<uint16_t> index_label_image;
        training_sample temp;
        while(data.is_enabled())
        {
            const image_info& image_info = listing[rnd.get_random_32bit_number()%listing.size()];
            load_image(input_image, image_info.image_filename);
            load_image(rgb_label_image, image_info.label_filename);
            rgb_label_image_to_index_label_image(rgb_label_image, index_label_image);
            randomly_crop_image(input_image, index_label_image, temp, rnd);
            data.enqueue(temp);
        }
    };
    std::thread data_loader1([f](){ f(1); });
    std::thread data_loader2([f](){ f(2); });
    std::thread data_loader3([f](){ f(3); });
    std::thread data_loader4([f](){ f(4); });

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    // We will run until the learning rate has dropped by a factor of 1e-3.
    while(trainer.get_learning_rate() >= initial_learning_rate*1e-4)
    {
        samples.clear();
        labels.clear();

        // make a 64 image mini-batch
        training_sample temp;
        while(samples.size() < 32)
        {
            data.dequeue(temp);

            samples.push_back(std::move(temp.first));
            labels.push_back(std::move(temp.second));
        }

        trainer.train_one_step(samples, labels);
    }

    // Training done, tell threads to stop and make sure to wait for them to finish before
    // moving on.
    data.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();

    // also wait for threaded processing to stop in the trainer.
    trainer.get_net();

    net.clean();
    cout << "saving network" << endl;
    serialize("squeezenetvoc2012.dnn") << net;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

