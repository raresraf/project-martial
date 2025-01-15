#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>
using namespace std;

/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/

int main() {
  string a;
  cin >> a;
  vector<char> vec;
  for (int i = 0; i < a.length(); i++) {
    vec.push_back(a.at(i));
  }
  sort(vec.begin(), vec.end());
  vec.erase(unique(vec.begin(), vec.end()), vec.end());
  if (vec.size() % 2 != 0) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }
  return 0;
}
