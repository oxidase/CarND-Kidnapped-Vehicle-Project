/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    static std::default_random_engine gen(std::random_device{}());
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 50;
    weights.resize(num_particles, 1.);
    particles.resize(num_particles);
    for (int i = 0; i < num_particles; ++i)
    {
        auto &p = particles[i];
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    static std::default_random_engine gen(std::random_device{}());
    std::normal_distribution<double> dist_x(0., std_pos[0]);
    std::normal_distribution<double> dist_y(0., std_pos[1]);
    std::normal_distribution<double> dist_theta(0., std_pos[2]);

    for (auto &p : particles)
    {
        const double x = p.x, y = p.y, theta = p.theta;

        if (std::fabs(yaw_rate) < 1e-6)
        { // CTRV model with zero yaw rate
            p.x = x + velocity * std::cos(theta) * delta_t + dist_x(gen);
            p.y = y + velocity * std::sin(theta) * delta_t + dist_y(gen);
            p.theta = theta + dist_theta(gen);
        }
        else
        { // CTRV model
            p.x = x + (velocity / yaw_rate) * ( std::sin(theta + yaw_rate * delta_t) - std::sin(theta)) + dist_x(gen);
            p.y = y + (velocity / yaw_rate) * (-std::cos(theta + yaw_rate * delta_t) + std::cos(theta)) + dist_y(gen);
            p.theta = theta + yaw_rate * delta_t + dist_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> &landmarks, const std::vector<LandmarkObs>& observations, const Map &map_landmarks)
{
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    if (map_landmarks.landmark_list.empty())
        return;

    for (auto &observation : observations)
    {
        auto nearest_neighbor = map_landmarks.landmark_list.begin();
        auto min_dist = dist(nearest_neighbor->x_f, nearest_neighbor->y_f, observation.x, observation.y);
        for (auto it = map_landmarks.landmark_list.begin() + 1; it != map_landmarks.landmark_list.end(); ++it)
        {
            auto d = dist(it->x_f, it->y_f, observation.x, observation.y);
            if (d < min_dist)
            {
                nearest_neighbor = it;
                min_dist = d;
            }
        }
        landmarks.push_back(*nearest_neighbor);
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks)
{
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    auto normal2d = [&std_landmark] (double x, double y, double mu_x, double mu_y) {
                        return std::exp(-(std::pow((x-mu_x) / std_landmark[0], 2) + std::pow((y-mu_y) / std_landmark[1], 2)) /2.)
                            / (2. * M_PI * std_landmark[0] * std_landmark[1]);
                    };


    for (int i = 0; i < num_particles; ++i)
    {
        auto &particle = particles[i];

        // Transform observations from vehicle to global coordinates
        // global        particle              local
        //  ┌ ┐   ┌                          ┐  ┌ ┐
        //  │x│ = │cos(theta)  -sin(theta)  x│  │x│
        //  │y│   │sin(theta)   cos(theta)  y│  │y│
        //  └ ┘   └                          ┘  │1│
        //                                      └ ┘
        std::vector<LandmarkObs> global_observations;
        for (const auto &local : observations)
        {
            LandmarkObs global;
            global.x = std::cos(particle.theta) * local.x - std::sin(particle.theta) * local.y + particle.x;
            global.y = std::sin(particle.theta) * local.x + std::cos(particle.theta) * local.y + particle.y;
            global_observations.push_back(global);
        }

        // Find nearest landmarks association
        std::vector<Map::single_landmark_s> nearest_landmarks;
        dataAssociation(nearest_landmarks, global_observations, map_landmarks);

        // Compute particle weight
        weights[i] = 1.;
        for (std::size_t j = 0; j < nearest_landmarks.size(); ++j)
        {
            weights[i] *= normal2d(global_observations[j].x, global_observations[j].y, nearest_landmarks[j].x_f, nearest_landmarks[j].y_f);
        }
    }

    // Normalize weights
    double sum = std::accumulate(weights.begin(), weights.end(), 0.);
    for (int i = 0; i < num_particles; ++i)
    {
        weights[i] = weights[i] / sum;
        particles[i].weight = weights[i];
    }
}


void ParticleFilter::resample()
{
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    static std::default_random_engine gen(std::random_device{}());
    std::vector<Particle> resampled_particles;
#if defined(USE_STD_RESAMPLER)
    std::discrete_distribution<> d(weights.begin(), weights.end());
    for (int i = 0; i < num_particles; ++i)
    {
        resampled_particles.push_back(particles[d(gen)]);
    }
#else
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::size_t idx;
    double beta, w_max;

    idx = std::uniform_int_distribution<std::size_t>(0, num_particles - 1)(gen);
    beta = 0.0;
    w_max = *std::max_element(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; ++i)
    {
        beta += uniform(gen) * 2.0 * w_max;
        while (beta > weights[idx])
        {
            beta -= weights[idx];
            idx = (idx + 1) % num_particles;
        }
        resampled_particles.push_back(particles[idx]);
    }
#endif
    std::swap(particles, resampled_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

std::string ParticleFilter::getAssociations(Particle best)
{
    std::vector<int> v = best.associations;
    std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseX(Particle best)
{
    std::vector<double> v = best.sense_x;
    std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseY(Particle best)
{
    std::vector<double> v = best.sense_y;
    std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
