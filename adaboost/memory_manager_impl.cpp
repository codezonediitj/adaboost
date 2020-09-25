#ifndef ADABOOST_MEMORY_MANAGER_IMPL_CPP
#define ADABOOST_MEMORY_MANAGER_IMPL_CPP

#include<set>
#include<string>
#include<climits>
#include<adaboost/utils/utils.hpp>
#include<adaboost/memory_manager.hpp>

namespace adaboost
{
    Base::~Base()
    {
    }

    MemoryManager* MemoryManager::MM = NULL;

    MemoryManager::MemoryManager() :
    current_memory(0), upper_bound(ULONG_MAX)
    {
    }

    MemoryManager* MemoryManager::create_MemoryManager()
    {
        if( MM != NULL )
        {
            return MM;
        }
        MM = new MemoryManager();
        return MM;
    }

    void MemoryManager::register_object(Base* obj)
    {
        std::string oom = std::to_string(this->upper_bound) + " bytes of memory have been exhausted.";
        adaboost::utils::check(this->current_memory + sizeof(obj) <= this->upper_bound, oom);
        this->ref_count.insert(obj);
        this->current_memory += sizeof(obj);
    }

    void MemoryManager::clear_object(Base* obj)
    {
        std::string unreg = "The object isn't registered yet, cannot clear the memory";
        adaboost::utils::check(this->ref_count.find(obj) != this->ref_count.end(), unreg);
        this->ref_count.erase(obj);
        this->current_memory -= sizeof(obj);
        delete obj;
    }

    void MemoryManager::clear_all()
    {
        for( Base* obj: this->ref_count )
        {
            delete obj;
        }
        this->ref_count.clear();
    }

    MemoryManager* memory_manager = MemoryManager::create_MemoryManager();

}

#endif
