#include <iostream>

using namespace std;

#define ll long long;

unsigned long long int n;

void solve() {
  string answer;
  bool lucky = false;
  int cont = 0;
  while (n > 0) {
    if (n % 10 == 4 || n % 10 == 7)
      cont++;
    n = (n - n % 10) / 10;
  }
  if (cont == 4 || cont == 7)
    lucky = true;
  answer = lucky ? "YES" : "NO";
  cout << answer;
}

int main() {
  cin >> n;
  solve();
}