#include <iostream>

using namespace std;

#define long long ll;

int n;
string operations[150];

void solve() {
  int x = 0;
  for (int i = 0; i < n; i++) {
    if (operations[i].find('+') != std::string::npos) {
      x++;
    }
    if (operations[i].find('-') != std::string::npos) {
      x--;
    }
  }
  cout << x;
}

int main() {
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> operations[i];
  }
  solve();
}