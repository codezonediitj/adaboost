#include<adaboost/tests/test_algorithm_naive_decision_stump.hpp>
#include<adaboost/tests/test_algorithm_discrete_adaboost.hpp>

int main(int ac, char* av[])
{
  testing::InitGoogleTest(&ac, av);
  return RUN_ALL_TESTS();
}
