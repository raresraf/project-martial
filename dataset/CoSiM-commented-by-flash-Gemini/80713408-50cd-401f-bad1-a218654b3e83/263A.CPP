#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int temp;
  pair<int, int> pos;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> temp;
      if (temp == 1) {
        pos.first = i;
        pos.second = j;
      }
    }
  }

  int res = abs(2 - pos.first) + abs(2 - pos.second);

  cout << res << endl;

  return 0;
}
