#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;

  for (int i = 0; i < s.size(); i++) {
    if (s.at(i) != 'A' && s.at(i) != 'a' && s.at(i) != 'E' && s.at(i) != 'e' &&
        s.at(i) != 'I' && s.at(i) != 'i' && s.at(i) != 'O' && s.at(i) != 'o' &&
        s.at(i) != 'U' && s.at(i) != 'u' && s.at(i) != 'Y' && s.at(i) != 'y') {
      if (s.at(i) > 'A' && s.at(i) <= 'Z')
        s.at(i) += 32;
      cout << "." << s.at(i);
    }
  }

  cout << endl;

  return 0;
}
