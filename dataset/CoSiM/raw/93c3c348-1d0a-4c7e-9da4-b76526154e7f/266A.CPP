#include <iostream>
#include <string>

using namespace std;

int main() {
  int letters;
  cin >> letters;
  string s;
  cin >> s;
  int remove_letters{0};
  for (size_t i = 1, j = 0; i < s.size(); ++i) {
    if (s[i] == s[j]) {
      ++remove_letters;
    } else {
      j = i;
    }
  }
  cout << remove_letters << endl;
  return 0;
}
