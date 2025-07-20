#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {

  // n will be useless, as the stones will be taken from user as a string
  short n;
  cin >> n;

  string stones;
  cin >> stones;

  short stones_to_remove{};

  for (short i = 0; i < stones.size() - 1; i++) {
    if (stones[i] == stones[i + 1]) {
      stones_to_remove++;
    }
  }

  cout << stones_to_remove;
}