/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem (e.g., Codeforces 236A),
 * demonstrating two approaches: one using std::set and a less efficient manual method.
 * @details The problem requires determining if the number of distinct characters in a
 * username is even or odd to classify the user as female ("CHAT WITH HER!") or
 * male ("IGNORE HIM!").
 */
#include <bits/stdc++.h>
using namespace std;

// The author notes their learning process, which is valuable semantic context.
// "Learned the concept of Sets in c++ , updated Solution using sets -"


// =================================================================================
// Active Solution: Using std::set (Efficient)
// =================================================================================
/**
 * @brief This is the primary, efficient solution presented in the file.
 * @algorithm Counting distinct elements using a `std::set`.
 * @complexity
 * - Time: O(L), where L is the string length. Each insertion into the set is
 *   O(log K), where K is the number of distinct characters (a small constant).
 * - Space: O(K), effectively O(1) as the character set is bounded.
 */
int main() {
  string s;
  cin >> s;

  // A set automatically stores only unique elements.
  set<char> un;
  
  // Insert each character from the string into the set.
  for (char ch : s) {
    un.insert(ch);
  }

  // The size of the set is the number of distinct characters.
  // Check if the count is even or odd.
  if (un.size() % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}


// =================================================================================
// Commented-out Solution: Manual distinct character counting (Inefficient)
// =================================================================================
/*
 * @brief This is a brute-force, less efficient implementation left as a comment.
 * @algorithm Manual iteration and checking for distinctness.
 * @complexity
 * - Time: O(L * K), where L is the string length and K is the number of distinct
 *   characters found so far. In the worst case, this approaches O(L^2).
 * - Space: O(K) to store the unique characters found.
 *
int main () {
  string s;
  cin>>s;

  // `c` will store the unique characters found.
  char c [101];
  c[0] = '\0';
  int currentsize=0;

  // For each character in the input string...
  for(char ch : s){
    bool flag = false;
    // ...search the array of already-found unique characters.
    for(int i = 0 ; i < currentsize; i++){ // Correctness note: should be i < currentsize
      if(ch == c[i]){
        flag = true;
        break;
      }
    }

    // If the flag is false, the character is new.
    if(flag == false) {
      c[currentsize] = ch;
      currentsize++;
    }
  }

  if(currentsize % 2 == 0) cout<<"CHAT WITH HER!"<<endl;
  else cout<<"IGNORE HIM!"<<endl;
}
*/
