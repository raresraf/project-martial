#include <iostream>
#include <string>

using namespace std;

int main() {

  string original, ret;

  cin >> original;

  int i;
  for (i = 0; i < original.length(); i++) {
    if (original[i] >= 65 && original[i] <= 90) {
      original[i] += 32;
    }
    if (original[i] != 'a' && original[i] != 'o' && original[i] != 'y' &&
        original[i] != 'e' && original[i] != 'u' && original[i] != 'i') {
      ret.push_back('.');
      ret.push_back(original[i]);
    }
  }

  cout << ret << "\n";

  return 0;
}
