#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  cin >> n;

  int out = 0;
  while (n--) {
    string s;
    cin >> s;
    if (s.front() == '-' || s.back() == '-')
      out--;
    if (s.front() == '+' || s.back() == '+')
      out++;
  }
  cout << out;
}