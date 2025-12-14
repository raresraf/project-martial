#include <iostream>

using namespace std;

#define long long ll;

string s;

void solve() {
  string answer;
  for (int i = 0; i < s.length(); i = i + 2) {
    for (int j = i; j < s.length(); j = j + 2) {
      if (s[i] > s[j]) {
        int aux = s[i];
        s[i] = s[j];
        s[j] = aux;
      }
    }
    if (i != s.length() - 1)
      cout << s[i] << "+";
    else
      cout << s[i];
  }
}

int main() {
  cin >> s;

  solve();
}