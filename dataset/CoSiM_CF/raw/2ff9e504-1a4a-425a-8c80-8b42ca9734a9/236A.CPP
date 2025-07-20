#include <iostream>
#include <set>
#include <string>
using namespace std;

int main() {

  string word;
  set<char> conj;
  cin >> word;
  char arr[word.size()] = {0};
  for (char &c : word)
    conj.insert(c);

  if (conj.size() % 2)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;
  return 0;
}
