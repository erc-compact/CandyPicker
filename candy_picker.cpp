#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <tuple>
#include <tinyxml2.h> // Include tinyxml2 for XML parsing
#include <filesystem>
#include <unordered_set>
#include <iomanip>
#include <getopt.h>
#include <cstdlib>  
#include <memory>

namespace fs = std::filesystem;
#define SPEED_OF_LIGHT 299792458.0


class Candidate;


struct CandidateHash {
    std::size_t operator()(const std::shared_ptr<Candidate>& c) const;
};


struct CandidateEqual {
    bool operator()(const std::shared_ptr<Candidate>& a, const std::shared_ptr<Candidate>& b) const;
};


std::vector<std::string> get_xml_files(const std::string& root_dir) {
    std::vector<std::string> xml_files;
    try {
        // Recursively iterate over the directory
        for (const auto& entry : fs::recursive_directory_iterator(root_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".xml") {
                xml_files.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error accessing files: " << e.what() << std::endl;
    }
    return xml_files;
}


class Candidate {
public:
    double snr;
    double period;
    double f0;
    double dm;
    double acc;
    int nh;
    float ddm_count_ratio;
    float ddm_snr_ratio;
    int nassoc;
    int period_ms;
    double pulse_width;
    std::string search_candidates_database_uuid;
    std::string xml_file_name;
    int candidate_id_in_file;
    bool is_pivot = false;
    std::vector<std::shared_ptr<Candidate>> related_candidates;

    Candidate()
        : snr(0),
          period(0),
          f0(0),
          dm(0),
          acc(0),
          nh(0),
          ddm_count_ratio(0),
          ddm_snr_ratio(0),
          nassoc(0),
          period_ms(0),
          pulse_width(0),
          search_candidates_database_uuid("") {}

    Candidate(double snr,
              double period,
              double dm,
              double acc,
              int nh,
              float ddm_count_ratio,
              float ddm_snr_ratio,
              int nassoc,
              const std::string& search_candidates_database_uuid,
              const std::string& xml_file_name,
              int candidate_id_in_file)
        : snr(snr),
          period(period),
          dm(dm),
          acc(acc),
          nh(nh),
          ddm_count_ratio(ddm_count_ratio),
          ddm_snr_ratio(ddm_snr_ratio),
          nassoc(nassoc),
          search_candidates_database_uuid(search_candidates_database_uuid),
          xml_file_name(xml_file_name),
          candidate_id_in_file(candidate_id_in_file)
    {
        f0 = 1.0 / period;
        period_ms = static_cast<int>(std::round(period * 1000.0));
        pulse_width = period / std::pow(2.0, nh);
    }

    bool operator<(const Candidate& other) const {
        return period < other.period;
    }

    bool operator==(const Candidate& other) const {
        // Must match how we define the hash
        // For example, compare these four parameters:
        return (std::fabs(period - other.period) < 1e-12) &&
               (std::fabs(dm - other.dm) < 1e-6) &&
               (std::fabs(acc - other.acc) < 1e-6) &&
               (nh == other.nh);
    }

    bool operator!=(const Candidate& other) const {
        return !(*this == other);
    }
    void make_pivot() {
        is_pivot = true;
    }

  
    bool is_related(const std::shared_ptr<Candidate>& other, double period_thresh, double tobs_over_c) const {
        const double dm_thresh = 100.0;
        if (std::fabs(dm - other->dm) > dm_thresh) return false;

        double corrected_other_period = 1/(other->f0 - (other->acc - acc) * other->f0 * tobs_over_c);
       
        double true_period_difference =
            (period / corrected_other_period) > 1.0
                ? std::fmod(period, corrected_other_period)
                : std::fmod(corrected_other_period, period);

        return true_period_difference <= period_thresh || 
               std::fabs(period - corrected_other_period) <= period_thresh;
        
    }

    void add_related(const std::shared_ptr<Candidate>& other) {
        related_candidates.push_back(other);
    }


}; 


std::size_t CandidateHash::operator()(const std::shared_ptr<Candidate>& c) const {

    auto h1 = std::hash<long long>()(static_cast<long long>(std::llround(c->period * 1e10)));
    auto h2 = std::hash<long long>()(static_cast<long long>(std::llround(c->dm     * 1e6)));
    auto h3 = std::hash<long long>()(static_cast<long long>(std::llround(c->acc    * 1e6)));
    auto h4 = std::hash<int>()(c->nh);

    auto combine = [](std::size_t seed, std::size_t v) {
        const std::size_t kMul = 0x9e3779b97f4a7c15ULL;
        seed ^= v + kMul + (seed << 6) + (seed >> 2);
        return seed;
    };

    std::size_t result = h1;
    result = combine(result, h2);
    result = combine(result, h3);
    result = combine(result, h4);

    return result;
}

bool CandidateEqual::operator()(const std::shared_ptr<Candidate>& a, const std::shared_ptr<Candidate>& b) const {
    return a == b;
}

class XMLFile {
    public:
        std::string filename;
        std::string misc_info;
        std::string header_parameters;
        std::string search_parameters;
        std::string segment_parameters;
        std::string dedispersion_trials;
        std::string acceleration_trials;
        std::string cuda_device_parameters;
        std::string execution_times;
        std::vector<std::shared_ptr<Candidate>> candidates;
        long long fft_size;
        double tsamp;

    void parse_xml_file(){
        try{
            tinyxml2::XMLDocument doc;
            if (doc.LoadFile(filename.c_str()) != tinyxml2::XML_SUCCESS) {
                throw std::runtime_error("Failed to load file: " + filename);
            }

            tinyxml2::XMLElement* root = doc.RootElement();
            if (!root) {
                throw std::runtime_error("No root element in XML for " + filename);
            }

            tinyxml2::XMLElement* misc_infoElem = root->FirstChildElement("misc_info");
            if (!misc_infoElem) {
                throw std::runtime_error("No <misc_info> element found in " + filename);
            }
            tinyxml2::XMLPrinter misc_info_printer;
            misc_infoElem->Accept(&misc_info_printer);
            misc_info = misc_info_printer.CStr();


            tinyxml2::XMLElement* header_parametersElem = root->FirstChildElement("header_parameters");
            if (!header_parametersElem) {
                throw std::runtime_error("No <header_parameters> element found in " + filename);
            }
            tinyxml2::XMLPrinter header_parameters_printer;
            header_parametersElem->Accept(&header_parameters_printer);
            header_parameters = header_parameters_printer.CStr();

            tsamp = std::stod(header_parametersElem->FirstChildElement("tsamp")->GetText());

            tinyxml2::XMLElement* search_parametersElem = root->FirstChildElement("search_parameters");
            if (!search_parametersElem) {
                throw std::runtime_error("No <search_parameters> element found in " + filename);
            }
            tinyxml2::XMLPrinter search_parameters_printer;
            search_parametersElem->Accept(&search_parameters_printer);
            search_parameters = search_parameters_printer.CStr();

            fft_size = std::stoll(search_parametersElem->FirstChildElement("size")->GetText());


            tinyxml2::XMLElement* segment_parametersElem = root->FirstChildElement("segment_parameters");
            if (!segment_parametersElem) {
                throw std::runtime_error("No <segment_parameters> element found in " + filename);
            }
            tinyxml2::XMLPrinter segment_parameters_printer;
            segment_parametersElem->Accept(&segment_parameters_printer);
            segment_parameters = segment_parameters_printer.CStr();

            tinyxml2::XMLElement* dedispersion_trialsElem = root->FirstChildElement("dedispersion_trials");
            if (!dedispersion_trialsElem) {
                throw std::runtime_error("No <dispersion_trials> element found in " + filename);
            }
            tinyxml2::XMLPrinter dedispersion_trials_printer;
            dedispersion_trialsElem->Accept(&dedispersion_trials_printer);
            dedispersion_trials = dedispersion_trials_printer.CStr();

            tinyxml2::XMLElement* acceleration_trialsElem = root->FirstChildElement("acceleration_trials");
            if (!acceleration_trialsElem) {
                throw std::runtime_error("No <acceleration_trials> element found in " + filename);
            }
            tinyxml2::XMLPrinter acceleration_trials_printer;
            acceleration_trialsElem->Accept(&acceleration_trials_printer);
            acceleration_trials = acceleration_trials_printer.CStr();


            tinyxml2::XMLElement* cuda_device_parametersElem = root->FirstChildElement("cuda_device_parameters");
            if (!cuda_device_parametersElem) {
                throw std::runtime_error("No <cuda_device_parameters> element found in " + filename);
            }
            tinyxml2::XMLPrinter cuda_device_parameters_printer;
            cuda_device_parametersElem->Accept(&cuda_device_parameters_printer);
            cuda_device_parameters = cuda_device_parameters_printer.CStr();

            tinyxml2::XMLElement* candidatesElem = root->FirstChildElement("candidates");
        for (tinyxml2::XMLElement* candidateElem = candidatesElem->FirstChildElement("candidate");
             candidateElem != nullptr;
             candidateElem = candidateElem->NextSiblingElement("candidate")) {
                // Safely get the text contents. If any child is missing, this throws an exception.
                const char* periodText                = candidateElem->FirstChildElement("period")->GetText();
                const char* dmText                    = candidateElem->FirstChildElement("dm")->GetText();
                const char* accText                   = candidateElem->FirstChildElement("acc")->GetText();
                const char* nhText                    = candidateElem->FirstChildElement("nh")->GetText();
                const char* snrText                   = candidateElem->FirstChildElement("snr")->GetText();
                const char* ddmCountRatioText         = candidateElem->FirstChildElement("ddm_count_ratio")->GetText();
                const char* ddmSnrRatioText           = candidateElem->FirstChildElement("ddm_snr_ratio")->GetText();
                const char* nassocText                = candidateElem->FirstChildElement("nassoc")->GetText();
                const char* uuidText                  = candidateElem->FirstChildElement("search_candidates_database_uuid")->GetText();
                const char* candidateId              = candidateElem->Attribute("id");

                if (!periodText || !dmText || !accText || !nhText || !snrText ||
                    !ddmCountRatioText || !ddmSnrRatioText || !nassocText || !uuidText || !candidateId)
                {
                    std::cerr << "xml has incomplete elements, skipping this one: " << filename << std::endl;
                    continue;
                }

                double period                 = std::stod(periodText);
                double dm                     = std::stod(dmText);
                double acc                    = std::stod(accText);
                int nh                        = std::stoi(nhText);
                double snr                    = std::stod(snrText);
                float ddm_count_ratio         = std::stof(ddmCountRatioText);
                float ddm_snr_ratio           = std::stof(ddmSnrRatioText);
                int nassoc                    = std::stoi(nassocText);
                std::string uuid              = uuidText;
                int candidate_id              = std::stoi(candidateId);

                std::shared_ptr<Candidate> c = std::make_shared<Candidate>(
                    snr, period, dm, acc,
                    nh, ddm_count_ratio, ddm_snr_ratio,
                    nassoc, uuid, filename, candidate_id
                );

                // Emplace a new Candidate
                candidates.emplace_back(c);
            }
            tinyxml2::XMLElement* execution_timesElem = root->FirstChildElement("execution_times");
            if (!execution_timesElem) {
                throw std::runtime_error("No <execution_times> element found in " + filename);
            }
            tinyxml2::XMLPrinter execution_times_printer;
            execution_timesElem->Accept(&execution_times_printer);
            execution_times = execution_times_printer.CStr();

            std::cout << "Parsed " << filename << std::endl;
            std::cout << "Found " << candidates.size() << " candidates in " << filename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing " << filename << ": " << e.what() << std::endl;
    }
    }


    XMLFile(const std::string& filename)
        : filename(filename) {
            parse_xml_file();
        }

    void write_updated_xmls(){
        std::string picked_filename = filename.substr(0, filename.find(".xml")) + "_picked.xml";
        std::string rejected_filename = filename.substr(0, filename.find(".xml")) + "_rejected.xml";
        std::ofstream picked(picked_filename);
        std::ofstream rejected(rejected_filename);
        if (!picked.is_open() || !rejected.is_open()) {
            std::cerr << "Error: Could not open new XML for writing.\n";
            return;
        }

        picked << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        rejected << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";

        picked << "<peasoup_search>\n";
        rejected << "<peasoup_search>\n";
        picked << misc_info << header_parameters << search_parameters << segment_parameters << dedispersion_trials << acceleration_trials << cuda_device_parameters;
        rejected << misc_info << header_parameters << search_parameters << segment_parameters << dedispersion_trials << acceleration_trials << cuda_device_parameters;
        picked << "<candidates>\n";
        rejected << "<candidates>\n";
        for (const auto& c : candidates) {
            std::stringstream ss;
            ss << "    <candidate id=\"" << c->candidate_id_in_file << "\">\n";
            ss << "      <period>" << c->period << "</period>\n";
            ss << "      <dm>" << c->dm << "</dm>\n";
            ss << "      <acc>" << c->acc << "</acc>\n";
            ss << "      <nh>" << c->nh << "</nh>\n";
            ss << "      <snr>" << c->snr << "</snr>\n";
            ss << "      <ddm_count_ratio>" << c->ddm_count_ratio << "</ddm_count_ratio>\n";
            ss << "      <ddm_snr_ratio>" << c->ddm_snr_ratio << "</ddm_snr_ratio>\n";
            ss << "      <nassoc>" << c->nassoc << "</nassoc>\n";
            ss << "      <search_candidates_database_uuid>" << c->search_candidates_database_uuid << "</search_candidates_database_uuid>\n";
            ss << "    </candidate>\n";
            c->is_pivot ? picked << ss.str() : rejected << ss.str();
        }
        picked << "</candidates>\n";
        rejected << "</candidates>\n";
        picked << execution_times;
        rejected << execution_times;

        picked << "</peasoup_search>\n";
        rejected << "</peasoup_search>\n";

        picked.close();
        rejected.close();
    }
};



void cluster_candidates(std::vector<std::shared_ptr<Candidate>>& candidates, double period_thresh, double tobs_over_c) {
    size_t n = candidates.size();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (candidates[i]->is_related(candidates[j], period_thresh, tobs_over_c)) {
                #pragma omp critical
                {
                    candidates[i]->add_related(candidates[j]);
                }
            }
        }
    }
    std::cout << "Finished clustering." << std::endl;
}


/**
 * Helper function that “shortlists” the pivot candidates (removes duplicates or
 * heavily-related ones) by marking everything in related_candidates for removal,
 * then removing them from the main list.
 */
std::vector<std::shared_ptr<Candidate>> shortlist_candidates(std::vector<std::shared_ptr<Candidate>>& candidates) {
    std::vector<std::shared_ptr<Candidate>> shortlisted_candidates;
    std::unordered_set<std::shared_ptr<Candidate>, CandidateHash, CandidateEqual> to_remove;

    // Step 1: Mark candidates to remove if they are in the "related" list of any
    // candidate that has more than 1 related candidate.
    #pragma omp parallel
    {
        // Thread-local sets & vectors
        std::unordered_set<std::shared_ptr<Candidate>, CandidateHash, CandidateEqual> local_to_remove;
        std::vector<std::shared_ptr<Candidate>> local_shortlisted;

        #pragma omp for nowait
        for (size_t i = 0; i < candidates.size(); ++i) {
            // If a candidate has more than 1 related candidate, remove the related ones
            if (candidates[i]->related_candidates.size() > 1) {
                for (const auto& related : candidates[i]->related_candidates) {
                    local_to_remove.insert(related);
                }
            }
            // Keep track of this candidate in our local shortlist
            local_shortlisted.push_back(candidates[i]);
        }

        // Merge thread-local results into global
        #pragma omp critical
        {
            to_remove.insert(local_to_remove.begin(), local_to_remove.end());
            shortlisted_candidates.insert(shortlisted_candidates.end(),
                                          local_shortlisted.begin(),
                                          local_shortlisted.end());
        }
    }
    std::shared_ptr<Candidate> null_candidate = std::make_shared<Candidate>();

    // Step 2: Filter out candidates marked for removal
    // Nullify them first
    #pragma omp parallel for
    for (size_t i = 0; i < shortlisted_candidates.size(); ++i) {
        if (to_remove.find(shortlisted_candidates[i]) != to_remove.end()) {
            shortlisted_candidates[i] =null_candidate;// Default => “null” marker
        }
    }


    // Remove the nullified ones
    shortlisted_candidates.erase(
        std::remove_if(shortlisted_candidates.begin(),
                       shortlisted_candidates.end(),
                       [null_candidate](const  std::shared_ptr<Candidate>& c) { return c == null_candidate; }),
        shortlisted_candidates.end());  


    //mark all pivots in original list if they are in shortlisted_candidates
    for (auto& c : candidates) {
        if (std::find(shortlisted_candidates.begin(), shortlisted_candidates.end(), c) != shortlisted_candidates.end()) {
            c->make_pivot();
        }
    }
    

    return shortlisted_candidates;
}



/**
 * Save a vector of Candidates to a CSV file, including how many are related, etc.
 */
void save_candidates_to_csv(const std::vector<std::shared_ptr<Candidate>>& pivots,
                            const std::string& filename) 
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }
    ofs << "snr,period,dm,acc,nh,ddm_count_ratio,ddm_snr_ratio,nassoc,"
        << "search_uuid,xml_file,candidate_id_in_file,num_related,related_cands\n";




    // Write data rows
    for (const auto& c : pivots) {
        ofs << c->snr << ","
            << std::fixed << std::setprecision(17) << c->period << "," << std::fixed << std::setprecision(8)
            << c->dm << ","
            << c->acc << ","
            << c->nh << ","
            << c->ddm_count_ratio << ","
            << c->ddm_snr_ratio << ","
            << c->nassoc << ","
            << c->period_ms << ","
            << c->search_candidates_database_uuid << ","
            << c->xml_file_name << ","
            << c->candidate_id_in_file << ","
            << c->related_candidates.size() << ",";
        // add related candidate uuids separated by :, take care of edge case
            if(c->related_candidates.size() > 0){
                ofs << c->related_candidates[0]->search_candidates_database_uuid;
                for(size_t i = 1; i < c->related_candidates.size(); i++){
                    ofs << ":" << c->related_candidates[i]->search_candidates_database_uuid;
                }
            }
            ofs << "\n";
        
        
    }

    ofs.close();
    std::cout << "Pivot candidates saved to " << filename << std::endl;
}


/**
 * Analyse the clustered candidates. In this example, we simply “shortlist” them,
 * sort them by the size of their related_candidate list, and write them to CSV.
 */
void save_pivots(std::vector<std::shared_ptr<Candidate>>& candidates) {
    // Shortlist
    std::vector<std::shared_ptr<Candidate>> shortlisted_candidates = shortlist_candidates(candidates);

    std::cout << "Found " << shortlisted_candidates.size() << " pivot candidates.\n";

    // Sort pivot candidates by number of related candidates, descending
    std::sort(shortlisted_candidates.begin(),
              shortlisted_candidates.end(),
              [](const std::shared_ptr<Candidate>& a, const std::shared_ptr<Candidate>& b) {
                  return a->related_candidates.size() > b->related_candidates.size();
              });


    save_candidates_to_csv(shortlisted_candidates, "pivots.csv");
}


int main(int argc, char** argv) {

    int opt;
    double period_thresh = 1e-7;
    double dm_thresh = 100;
    int ncpus = 8;
    std::vector<std::string> xml_files;

    // Parse options
    while ((opt = getopt(argc, argv, "p:d:n:")) != -1) {
        switch (opt) {
            case 'p':  // Period threshold
                period_thresh = std::atof(optarg);
                break;
            case 'd':  // DM threshold
                dm_thresh = std::atof(optarg);
                break;
            case 'n': // ncpus
                ncpus = std::atoi(optarg);
                break;
            case '?':  // Unknown option
                std::cerr << "Usage: " << argv[0] << " -p <period> -d <dm_thresh> -n <ncpus> file1.xml file2.xml ...\n";
                return 1;
        }
    }

    // Remaining arguments after option parsing are XML files
    for (int i = optind; i < argc; ++i) {
        xml_files.push_back(argv[i]);
    }

    // Validate input
    if (xml_files.empty()) {
        std::cerr << "Error: No XML files provided.\n";
        std::cerr << "Usage: " << argv[0] << " -p <period> -d <dm_thresh> file1.xml file2.xml ...\n";
        return 1;
    }

    // Print parsed values
    std::cout << "Period Threshold: " << period_thresh << "\n";
    std::cout << "DM Threshold: " << dm_thresh << "\n";
    std::cout << "XML files provided:\n";
    for (const auto& file : xml_files) {
        std::cout << " - " << file << '\n';
    }

    omp_set_num_threads(ncpus);


    std::vector<std::shared_ptr<Candidate>> all_candidates;
    std::vector<XMLFile> xml_file_objects;

    #pragma omp parallel
    {
        std::vector<std::shared_ptr<Candidate>> local_candidates;
        std::vector<XMLFile> local_xml_files;
        #pragma omp for
        for (size_t i = 0; i < xml_files.size(); ++i) {
            local_xml_files.emplace_back(XMLFile(xml_files[i]));
            local_candidates.insert(local_candidates.end(),
                                    local_xml_files.back().candidates.begin(),
                                    local_xml_files.back().candidates.end());            
        }


        #pragma omp critical
        {
            xml_file_objects.insert(xml_file_objects.end(),
                                    local_xml_files.begin(),
                                    local_xml_files.end());

            all_candidates.insert(all_candidates.end(),
                                  local_candidates.begin(),
                                  local_candidates.end());

        }
    }


    std::sort(all_candidates.begin(),
              all_candidates.end(),
              [](const std::shared_ptr<Candidate>& a, const std::shared_ptr<Candidate>& b) {
                  return a->snr > b->snr;
              });

    if (all_candidates.empty()) {
        std::cerr << "No candidates found in XML files.\n";
        return 1;
    }

    if(xml_file_objects.size() > 1) {
        for(size_t i = 1; i < xml_file_objects.size(); i++){
            if(xml_file_objects[i].fft_size != xml_file_objects[0].fft_size || xml_file_objects[i].tsamp != xml_file_objects[0].tsamp){
                std::cerr << "Error: fft size and tsamp are not the same for all files" << std::endl;
                return 1;
            }
        }
    }

    double effective_tobs = xml_file_objects[0].fft_size * xml_file_objects[0].tsamp;
    std::cout << "Effective TOBS: " << effective_tobs << " seconds" << std::endl;
    double tobs_over_c = effective_tobs  / SPEED_OF_LIGHT;


    cluster_candidates(all_candidates, period_thresh, tobs_over_c);
    save_pivots(all_candidates);

    #pragma omp parallel for
    for (size_t i = 0; i < xml_file_objects.size(); ++i) {
        xml_file_objects[i].write_updated_xmls();
    }


    return 0;
}
