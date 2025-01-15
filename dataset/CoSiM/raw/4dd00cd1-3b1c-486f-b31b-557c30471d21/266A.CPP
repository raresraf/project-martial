// CodeForces
// Problem 266A [Stones on the Table]

#include <iostream>
using namespace std;

int main() {
  int n, counter = 0;
  char str[50];

  cin >> n;
  cin >> str[0];

  for (int i = 1; i < n; i++) {
    cin >> str[i];
    if (str[i] == str[i - 1]) {
      counter++;
    }
  }
  cout << counter;
  return 0;
}