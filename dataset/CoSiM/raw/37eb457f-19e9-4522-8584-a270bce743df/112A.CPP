#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  string line1, line2;
  cin >> line1 >> line2;

  if (line1.size() != line2.size()) {
    return -1;
  }

  for (decltype(line1.size()) i = 0; i != line1.size(); ++i) {
    if (tolower(line1[i]) < tolower(line2[i])) {
      cout << "-1";
      return 0;
    } else if (tolower(line1[i]) > tolower(line2[i])) {
      cout << "1";
      return 0;
    }
  }

  cout << "0";
  return 0;
}
