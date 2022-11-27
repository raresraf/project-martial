#include <iostream>

using namespace std;

int main() {
  int w;
  cin >> w; // Weight of the Watermelon

  if (w >= 1 && w <= 100) {
    if (w != 2 && w % 2 == 0)
      cout << "YES";
    else
      cout << "NO";
  } else
    cout << "Enter a valid input";
  return 0;
}
