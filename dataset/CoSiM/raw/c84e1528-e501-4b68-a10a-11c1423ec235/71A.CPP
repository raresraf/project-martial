#include <iostream>
#include <limits>
#include <string>

using namespace std;

int main() {
  int count{};
  cin >> count;
  cin.ignore(numeric_limits<streamsize>::max(), '\n'); /* cleaning the buffer */

  for (int i = 0; i < count; i++) {
    string name{};
    getline(cin, name);

    int length{};
    length = name.size();

    if (length <= 10)
      cout << name << '\n';
    else
      cout << name[0] << length - 2 << name[length - 1] << '\n';
  }

  return 0;
}