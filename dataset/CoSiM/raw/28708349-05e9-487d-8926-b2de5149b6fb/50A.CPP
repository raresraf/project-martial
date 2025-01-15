#include <iostream>

using namespace std;

int main() {
  int n, m;
  cin >> n >> m;
  cout << (m - m % 2) * (n - n % 2) / 2 + m * (n % 2) / 2 + n * (m % 2) / 2;
}