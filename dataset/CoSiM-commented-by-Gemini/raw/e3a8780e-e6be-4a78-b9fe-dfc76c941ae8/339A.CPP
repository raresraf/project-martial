#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string input, output;
  cin >> input;
  int count1 = 0, count2 = 0, count3 = 0;
  for (int i = 0; i < input.length(); i++) {
    if (input[i] == '1')
      count1++;
    else if (input[i] == '2')
      count2++;
    else if (input[i] == '3')
      count3++;
  }
  int i;
  for (i = 0; i < count1 && count1 > 0; i++)
    output += "1+";
  for (i = 0; i < count2 && count2 > 0; i++)
    output += "2+";
  for (i = 0; i < count3 && count3 > 0; i++)
    output += "3+";
  cout << output.substr(0, output.length() - 1);
  return 0;
}
