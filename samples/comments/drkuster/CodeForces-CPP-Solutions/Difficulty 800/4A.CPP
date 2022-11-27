#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int watermelonWeight = 0;
  cin >> watermelonWeight;

  int originalWeight = watermelonWeight;

  bool shouldReturnYes = false;

  watermelonWeight -= 1;

  while (watermelonWeight > 0) {
    int otherHalf = originalWeight - watermelonWeight;

    if ((watermelonWeight % 2 == 0) && (otherHalf % 2 == 0)) {
      shouldReturnYes = true;
      break;
    }

    watermelonWeight -= 1;
  }

  if (shouldReturnYes) {
    cout << "YES" << endl;
  }

  else {
    cout << "NO" << endl;
  }

  return 0;
}