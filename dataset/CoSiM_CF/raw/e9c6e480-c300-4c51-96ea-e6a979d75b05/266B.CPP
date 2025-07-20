#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, t;
  string S;
  cin >> n >> t;
  cin >> S;
  for (int i = 0; i < t; i++)
    for (int j = 0; j + 1 < n; j++) {
      if (S[j] == 'B' && S[j + 1] == 'G') {
        S[j] = 'G';
        S[j + 1] = 'B';
        j++;
      }
    }
  cout << S;
  return 0;
}