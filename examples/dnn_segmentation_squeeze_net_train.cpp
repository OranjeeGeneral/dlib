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

typedef std::pair<matrix<rgb_pixel>, matrix<uint16_t>> training_sample;

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
                                                    bn_con<cont<18,3,3,2,2,
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
    auto size = 259;
    rectangle rect(size, size);
    // randomly shift the box around
	int xOffset = 0;
	int yOffset = 0;
	if (rect.width() < img.nc())
	{
		xOffset = rnd.get_random_32bit_number() % (img.nc() - rect.width());
	}
	if (rect.height() < img.nr())
	{
		yOffset = rnd.get_random_32bit_number() % (img.nr() - rect.height());
	}
    point offset(xOffset,yOffset);

    return move_rect(rect, offset);
}

std::vector<image_info> get_segmentation_training_data(const std::string& image_folder,
													   const std::string&  seg_image_folder)
{
	std::vector<image_info> results;
	// load normal images first
	{
		auto subdir = directory(image_folder);
		image_info temp;
		// Now get all the images in the directory
		for (auto image_file : subdir.get_files())
		{
			temp.image_filename = image_file;
			results.push_back(temp);
		}
	}
	int index = 0;
	// load segmentation label images next
	{
		auto subdir = directory(seg_image_folder);
		for (auto image_file : subdir.get_files())
		{
			if (index < results.size())
			{
				results[index].label_filename = image_file;
				index++;
			}
			else {
				cerr << "Error found more label images than real images\n";
			}
		}
	}
	if (index < results.size())
	{
		cerr << "Found less label images than real images\n";
	}
	return results;
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

    const chip_details chip_details(rect, chip_dims(259, 259));

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
}

// --------------------------------------------------------------------------------------


int main(int argc, char** argv) try
{
    if (argc != 3)
    {
		cerr << "Wrong number of arguments\n";
        return 1;
    }

    cout << "\nSCANNING DATASET\n" << endl;

    const auto listing = get_segmentation_training_data(argv[1],argv[2]);
    cout << "images in dataset: " << listing.size() << endl;
    if (listing.size() == 0)
    {
        cout << "Didn't find any training data " << endl;
        return 1;
    }


    set_dnn_prefer_smallest_algorithms();


    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    net_type net;

    dnn_trainer<net_type> trainer(net,sgd(weight_decay, momentum));
    trainer.be_verbose();
    trainer.set_learning_rate(initial_learning_rate);
    trainer.set_synchronization_file("squeeezesegnet_trainer_state_file.dat", std::chrono::minutes(10));

	// This threshold is probably excessively large.  You could likely get good results
    // with a smaller value but if you aren't in a hurry this value will surely work well.
    
	trainer.set_iterations_without_progress_threshold(10000);
    // Since the progress threshold is so large might as well set the batch normalization
    // stats window to something big too.
    
	set_all_bn_running_stats_window_sizes(net, 500);

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
            load_image(index_label_image, image_info.label_filename);
            randomly_crop_image(input_image, index_label_image, temp, rnd);
			if (temp.first.nc() < 256 || temp.first.nr() < 256)
				cout << "Input Size = " << temp.first.nc() << " " << temp.first.nr() << " " << temp.second.nc() << " " << temp.second.nr() << "\n";
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
    serialize("squeezenet_seg.dnn") << net;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

