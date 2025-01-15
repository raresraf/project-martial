#include <iostream>

using namespace std;

int main() {
  int numOfStones;
  int numToRemove = 0;
  string stones;
  cin >> numOfStones;
  cin >> stones;
  for (int i = 1; i < numOfStones; i++) {
    if (stones[i - 1] == stones[i]) {
      numToRemove += 1;
    }
  }
  cout << numToRemove << endl;
  return 0;
}