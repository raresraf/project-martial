#include <iostream>

#include <string>
#include <vector>

#include <algorithm>

/*
 * Contest: Code Forces Round #197 (Task 339-A)
 * URL: http://codeforces.ru/contest/339/problem/A
 */

int main(int argc, char **argv) {
  std::string s;
  std::vector<char> v;
  std::cin >> s;

  for (int i = 0; i < s.length(); i += 2) {
    v.push_back(s[i]);
  }

  std::sort(v.begin(), v.end());

  for (int i = 0; i < s.length(); i += 2) {
    s[i] = v[i >> 1];
  }

  std::cout << s << std::endl;
  return 0;
}
