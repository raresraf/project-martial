#include <iostream>
#include <string>

using namespace std;

string toLowerCase(string s) {
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] - 'a' < 0)
      s[i] += 32;
  }
  return s;
}

int compare(string first, string second) {
  first = toLowerCase(first);
  second = toLowerCase(second);
  if (first < second)
    return -1;
  return first == second ? 0 : 1;
}

int main() {
  string first, second;
  cin >> first >> second;
  cout << compare(first, second) << endl;
  return 0;
}
