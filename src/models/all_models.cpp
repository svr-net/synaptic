// The list of every model of the library (explicit, so a static library keeps them all).
#include "syn/kernel/kernel.hpp"

namespace syn::models {
const kernel::Model* spaceModel();
const kernel::Model* conditioningModel();
const kernel::Model* operantModel();
const kernel::Model* tokenEconomyModel();
}  // namespace syn::models

namespace syn::kernel {

std::vector<const Model*> allModels() {
  using namespace syn::models;
  return {spaceModel(), conditioningModel(), operantModel(), tokenEconomyModel()};
}

}  // namespace syn::kernel
