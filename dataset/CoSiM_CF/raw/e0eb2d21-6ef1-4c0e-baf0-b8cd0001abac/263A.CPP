#include <iostream>
using namespace std;
int main() {

  int sv;
  int step;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> sv;
      sv == 1 ? step = abs(2 - i) + abs(2 - j) : sv;
    }
  }

  cout << step;

  return 0;
}
