#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int v[5];
  int i = 0;
  int flag;
  do {
    for (int j = 0; j < 5; j++) {
      std::cin >> v[j];
      if (v[j] == 1) {
        flag = abs(i - 2) + abs(j - 2);
      }
    }
    i++;
  } while (i < 5);

  cout << flag;
  return 0;
}
