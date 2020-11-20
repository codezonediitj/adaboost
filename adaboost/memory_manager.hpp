#ifndef ADABOOST_MEMORY_MANAGER_HPP
#define ADABOOST_MEMORY_MANAGER_HPP

#include<unordered_set>

namespace adaboost
{
    /*
    * Base class for all the objects in adaboost namespace.
    */
    class Base
    {

        public:

            //! Virtual destructor
            virtual ~Base();

    };

    /*
    * Tracks each creation and deletion of each object under adaboost namespace.
    */
    class MemoryManager
    {

        private:

            //! Address of the only object created under singleton pattern.
            static MemoryManager* MM;

            //! Set of addresses of objects created.
            std::unordered_set<Base*> ref_count;

            //! The maximum memory that can be occupied by all the objects under adaboost namespace.
            unsigned long upper_bound;

            //! Current memory occupied by all the objects under adaboost namespace.
            unsigned long current_memory;

            //! Private constructor.
            MemoryManager();

        public:

            //! Creates MemoryManager object if non-existent otherwise
            //! returns the address of the previously created object.
            static MemoryManager* create_MemoryManager();

            /*
            * Registers the given object created under adaboost namespace.
            *
            * @param obj The address of the object to be registered.
            */
            void register_object(Base* obj);

            /*
            * Clears  object created under adaboost namespace.
            *
            * @param obj The address of the object to be cleared.
            */
            void clear_object(Base* obj);

            //! Clears all the objects created so far.
            void clear_all();

            /*
            * Resets the upper bound of memory that can be occupied by all the objects
            * under adaboost namespace.
            *
            * @param upper_bound A positive integer.
            */
            void set_upper_bound(unsigned long upper_bound);

    };

    extern MemoryManager* memory_manager;

}

#endif
