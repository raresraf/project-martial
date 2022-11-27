#include <iostream>

using namespace std;

int main() {
  int w, upperHalf, lowerHalf;

  cin >> w;

  upperHalf = lowerHalf = w / 2;
  if (w % 2)
    upperHalf++;
  while (upperHalf < w - 1) {
    if (upperHalf % 2 == 0 && lowerHalf % 2 == 0) {
      cout << "YES" << endl;
      break;
    }
    upperHalf++;
    lowerHalf--;
  }
  if (upperHalf > w - 2)
    cout << "NO" << endl;

  return 0;
}