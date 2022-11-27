#include <iostream>

using namespace std;

int main() {
  int watermalon_weight;
  cin >> watermalon_weight;
  if (watermalon_weight > 2 && watermalon_weight % 2 == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}