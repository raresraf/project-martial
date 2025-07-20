#include <algorithm>
#include <stdio.h>

int main() {
  int num_stops, max_people{0}, capacity{0};
  scanf("%d", &num_stops);
  for (int num_stop = 0; num_stop < num_stops; ++num_stop) {
    int leave, enter;
    scanf("%d%d", &leave, &enter);
    capacity = capacity + enter - leave;
    max_people = std::max(max_people, capacity);
  }
  printf("%d", max_people);
  return 0;
}
