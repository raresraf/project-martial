#include <iostream>

using namespace std;

int find(long long n) {
  int ans = 0;
  while (n != 0) {
    if (n % 10 == 4 || n % 10 == 7)
      ++ans;
    n /= 10;
  }
  return ans;
}

void lucky_digits(int n) {
  if (n == 4 || n == 7)
    cout << "YES";
  else
    cout << "NO";
}

int main() {
  long long n;
  cin >> n;
  lucky_digits(find(n));
}