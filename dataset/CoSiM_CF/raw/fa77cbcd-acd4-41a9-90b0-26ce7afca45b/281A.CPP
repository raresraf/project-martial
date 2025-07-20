#include <iostream>
using namespace std;

int main() {
  string word;
  cin >> word;
  cout << char(toupper(word[0]));
  int n = word.length();
  for (int i = 1; i < n; i++)
    cout << word[i];
  cout << "\n";
}
