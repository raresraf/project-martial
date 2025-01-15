#include <iostream>

using namespace std;

int main() {
  int n;
  int x = 0;
  cin >> n;

  for (int i = 0; i < n; i++) {
    string command;
    cin >> command;
    if (command.compare("X++") == 0 || command.compare("++X") == 0) {
      x = x + 1;
    }
    if (command.compare("X--") == 0 || command.compare("--X") == 0) {
      x = x - 1;
    }
  }
  cout << x << "\n";
  return 0;
}