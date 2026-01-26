#include <algorithm>
#include <iostream>
#include <sstream>
using namespace std;

int main() {
  string input, temp;
  cin >> input;

  int len = input.length() / 2 + 1;
  int data[len], k = 0;

  for (int i = 0; i < input.length(); i++)
    if (i % 2 == 0) {
      temp = input.at(i);
      stringstream ss(temp);
      ss >> data[k++];
      temp = "";
    }

  int n = sizeof(data) / sizeof(data[0]);
  sort(data, data + n);

  for (int i = 0; i < len; i++)
    if (i == len - 1)
      cout << data[i] << endl;
    else
      cout << data[i] << '+';

  return 0;
}