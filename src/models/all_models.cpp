// The list of every model of the library (explicit, so a static library keeps them all).
#include "syn/kernel/kernel.hpp"

namespace syn::models {
const kernel::Model* spaceModel();
const kernel::Model* conditioningModel();
const kernel::Model* operantModel();
const kernel::Model* tokenEconomyModel();
const kernel::Model* maslowModel();
const kernel::Model* ispModel();
const kernel::Model* memoryModel();
const kernel::Model* interleavingModel();
const kernel::Model* schemaModel();
const kernel::Model* equilibrationModel();
const kernel::Model* zpdModel();
const kernel::Model* dialogueModel();
const kernel::Model* developmentModel();
const kernel::Model* andragogyModel();
const kernel::Model* motivationModel();
const kernel::Model* classroomModel();
const kernel::Model* theoriesModel();
const kernel::Model* allocationModel();
const kernel::Model* sensitivityModel();
const kernel::Model* gagneModel();
const kernel::Model* audienceModel();
const kernel::Model* communityModel();
const kernel::Model* outcomesModel();
const kernel::Model* assessmentActivitiesModel();
const kernel::Model* assessmentSessionModel();
const kernel::Model* rubricModel();
const kernel::Model* lectureModel();
const kernel::Model* discussionModel();
const kernel::Model* scaffoldingModel();
const kernel::Model* strategiesModel();
const kernel::Model* lessonPlanModel();
const kernel::Model* reflectionModel();
const kernel::Model* studentFeedbackModel();
const kernel::Model* criticalReflectionModel();
const kernel::Model* professionalGrowthModel();
const kernel::Model* fgaAlgebraModel();
const kernel::Model* fgaLearningModel();
const kernel::Model* fgaRotorsModel();
}  // namespace syn::models

namespace syn::kernel {

std::vector<const Model*> allModels() {
  using namespace syn::models;
  return {spaceModel(), conditioningModel(), operantModel(), tokenEconomyModel(), maslowModel(), ispModel(), memoryModel(), interleavingModel(), schemaModel(), equilibrationModel(), zpdModel(), dialogueModel(), developmentModel(), andragogyModel(), motivationModel(), classroomModel(), theoriesModel(), allocationModel(), sensitivityModel(), gagneModel(), audienceModel(), communityModel(), outcomesModel(), assessmentActivitiesModel(), assessmentSessionModel(), rubricModel(), lectureModel(), discussionModel(), scaffoldingModel(), strategiesModel(), lessonPlanModel(), reflectionModel(), studentFeedbackModel(), criticalReflectionModel(), professionalGrowthModel(), fgaAlgebraModel(), fgaLearningModel(), fgaRotorsModel()};
}

}  // namespace syn::kernel
