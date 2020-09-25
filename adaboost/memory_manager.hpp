#ifndef ADABOOST_MEMORY_MANAGER_HPP
#define ADABOOST_MEMORY_MANAGER_HPP

#include<unordered_set>

namespace adaboost
{
    class Base
    {

        public:

            virtual ~Base();

    };

    class MemoryManager
    {

        private:

            static MemoryManager* MM;

            std::unordered_set<Base*> ref_count;

            unsigned long upper_bound;

            unsigned long current_memory;

            MemoryManager();

        public:

            static MemoryManager* create_MemoryManager();

            void register_object(Base* obj);

            void clear_object(Base* obj);

            void clear_all();

            void set_upper_bound(int upper_bound);

    };

    extern MemoryManager* memory_manager;

}

#endif
