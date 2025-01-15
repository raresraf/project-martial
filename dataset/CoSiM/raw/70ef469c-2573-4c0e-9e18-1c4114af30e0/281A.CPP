#include <bits/stdc++.h>
using namespace std;
int main() {
  int number;
  cin >> number;
  int X = 0;
  while (number > 0) {
    string st;
    cin >> st;
    if (st.at(1) == '+')
      X++;
    else
      X--;
    number--;
  }
  cout << X;
}