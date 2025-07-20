#include <iostream>
#include <vector>

using namespace std;

int main() {
  int size = 5;
  int answer;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int cell;
      cin >> cell;
      if (cell) {
        answer = abs(i - size / 2) + abs(j - size / 2);
      }
    }
  }
  cout << answer << endl;
  return 0;
}
