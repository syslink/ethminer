/// OpenCL miner implementation.
///
/// @file
/// @copyright GNU General Public License

#pragma once

#include <fstream>

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wmissing-braces"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS true
#define CL_HPP_ENABLE_EXCEPTIONS true
#define CL_HPP_CL_1_2_DEFAULT_BUILD true
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/cl2.hpp"
#pragma GCC diagnostic pop

// macOS OpenCL fix:
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
#endif

#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
#endif

namespace dev
{
namespace eth
{
// WARNING: Do not change the value of the following constant
// unless you are prepared to make the neccessary adjustments
// to the assembly code for the binary kernels.
const size_t c_maxSearchResults = 15;
// NOTE: The following struct must match the one defined in
// ethash.cl
struct SearchResults
{
    struct
    {
        uint32_t gid;
        // Can't use h256 data type here since h256 contains
        // more than raw data. Kernel returns raw mix hash.
        uint32_t mix[8];
        uint32_t pad[7];  // pad to 16 words for easy indexing
    } rslt[c_maxSearchResults];
    uint32_t count;
    uint32_t hashCount;
    uint32_t abort;
};

class CLMiner : public Miner
{
public:

    CLMiner(unsigned _index, CLSettings _settings, DeviceDescriptor& _device);
    ~CLMiner() override;

    static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);

protected:
    bool initDevice() override;

    bool initEpoch_internal() override;

    void kick_miner() override;

private:
    void workLoop() override;
    bool dagPrepare();
    void setSearchArgs(const WorkPackage& w);
    void search(uint64_t nonce);
    void readResult(SearchResults& results);
    void handleResult(const WorkPackage& current, const SearchResults& results);
    void sleep(int second);

    vector<cl::Context> m_context;
    vector<cl::CommandQueue> m_queue;
    vector<cl::CommandQueue> m_abortqueue;
    cl::Kernel m_searchKernel;
    cl::Kernel m_dagKernel;
    cl::Device m_device;

    vector<cl::Buffer> m_dag;
    vector<cl::Buffer> m_light;
    vector<cl::Buffer> m_header;
    vector<cl::Buffer> m_searchBuffer;

    CLSettings m_settings;

    unsigned m_dagItems = 0;
    uint64_t m_lastNonce = 0;

};

}  // namespace eth
}  // namespace dev
