#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int s;
  string s1, s2;
  cin >> s1;
  cin >> s2;
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  if (s1.compare(s2) == 0) {
    s = 0;
  } else if (s1.compare(s2) > 0) {
    s = 1;
  } else {
    s = -1;
  }
  cout << s;
  return 0;
}
