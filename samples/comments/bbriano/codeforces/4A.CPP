#include <iostream>
using namespace std;

int main() {
  int w;
  cin >> w;
  for (int i = 2; i < w; i += 2) {
    if (((w - i) % 2) == 0) {
      cout << "YES\n";
      return 0;
    }
  }
  cout << "NO\n";
}
