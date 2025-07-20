#include <iostream>

typedef long long int ll;

bool lucky(int n) { // since n <= 19
  return ((n == 4) || (n == 7));
}

bool nearlyLucky(std::string s) {
  ll count = 0;
  for (int i = 0; i < s.size(); ++i) {
    if ((s[i] == '4') || (s[i] == '7'))
      ++count;
  }
  return lucky(count);
}

int main() {
  std::string s;
  std::cin >> s;
  std::cout << (nearlyLucky(s) ? "YES" : "NO") << '\n';
}
