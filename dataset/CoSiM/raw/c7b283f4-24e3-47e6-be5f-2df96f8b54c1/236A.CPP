#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

bool check(string s) {
  vector<char> vec;
  for (char c : s) {
    if (find(vec.begin(), vec.end(), c) ==
        vec.end()) // If cannot find duplicate character in vec.
      vec.push_back(c);
  }
  if (vec.size() % 2 == 0)
    return true;
  else
    return false;
}

int main() {
  string name;
  while (cin >> name) {
    if (check(name))
      cout << "CHAT WITH HER!" << endl;
    else
      cout << "IGNORE HIM!" << endl;
  }
  return 0;
}
