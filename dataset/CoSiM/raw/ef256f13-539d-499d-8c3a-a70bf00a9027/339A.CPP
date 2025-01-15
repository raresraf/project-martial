#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>
using namespace std;
int main() {
  /*int i;
  string input;
  getline(cin,input);
      vector <char> numbers;
      for(i = 0; i < input.length(); i= i + 2)
      {
          numbers.push_back(input[i]);
      }
      sort(numbers.begin(),numbers.begin() + numbers.size());
      for(i = 0; i < numbers.size() - 1; i++)
      {
          cout << numbers[i] << "+";
      }
      cout << numbers[i] << endl;
*/
  string ch;
  int i, j;
  cin >> ch;
  vector<char> numbers;
  for (i = 0; i < ch.length(); i = i + 2) {
    numbers.push_back(ch[i]);
  }
  sort(numbers.begin(), numbers.begin() + numbers.size());
  for (i = 0; i < numbers.size() - 1; i++) {
    cout << numbers[i] << "+";
  }
  cout << numbers[i] << endl;
  return 0;
}
