#include <iostream>

using namespace std;

int main() {
  int num_tasks;
  cin >> num_tasks;

  int solutions{0};
  for (int num_task = 0; num_task < num_tasks; ++num_task) {
    int petya, vasya, tonya;
    cin >> petya >> vasya >> tonya;
    solutions += (petya + vasya + tonya) > 1 ? 1 : 0;
  }
  cout << solutions << endl;
  return 0;
}
