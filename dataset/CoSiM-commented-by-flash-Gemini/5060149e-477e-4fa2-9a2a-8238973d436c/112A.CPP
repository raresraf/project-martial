#include <bits/stdc++.h>
using namespace std;

int main() {
  string input1, input2;
  cin >> input1;
  cin >> input2;
  transform(input1.begin(), input1.end(), input1.begin(), ::tolower);
  transform(input2.begin(), input2.end(), input2.begin(), ::tolower);
  if (input1 == input2) {
    cout << "0" << endl;
  } else if (input1 > input2) {
    cout << "1" << endl;
  } else {
    cout << "-1" << endl;
  }

  return 0;
}
