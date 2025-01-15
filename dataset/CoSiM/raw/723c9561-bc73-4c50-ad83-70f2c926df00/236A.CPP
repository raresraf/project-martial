#include <iostream>
#include <set>
#include <string>

using namespace std;

int main() {
  string input;
  cin >> input;
  set<char> charSet(input.begin(), input.end());
  cout << ((charSet.size() % 2 == 0) ? "CHAT WITH HER!" : "IGNORE HIM!")
       << endl;
}