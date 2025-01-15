#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  char inp[n];
  cin >> inp;

  char last = inp[0];
  int counter = 0;

  for (int i = 1; i < n; i++) {
    if (inp[i] == 'R') {
      if (last == 'R')
        counter++;
      else
        last = 'R';
    }
    if (inp[i] == 'G') {
      if (last == 'G')
        counter++;
      else
        last = 'G';
    }
    if (inp[i] == 'B') {
      if (last == 'B')
        counter++;
      else
        last = 'B';
    }
  }

  cout << counter;
}
