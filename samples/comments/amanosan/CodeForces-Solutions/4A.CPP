#include <iostream>

using namespace std;

int main() {
  int w;
  cin >> w;
  if (w >= 4 && w % 2 == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}